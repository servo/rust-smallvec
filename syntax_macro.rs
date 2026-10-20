use proc_macro::{
    Delimiter,
    TokenStream,
    TokenTree
};

#[proc_macro_attribute]
pub fn syntax(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut item_attrs = Vec::<TokenTree>::new();
    let mut iter = item.into_iter().fuse();
    loop {
        match [iter.next(), iter.next()] {
            [Some(TokenTree::Punct(p)), Some(TokenTree::Group(g))]
                if (p.as_char(), g.delimiter()) == ('#', Delimiter::Bracket) =>
            {
                item_attrs.extend([p.into(), g.into()]);
            }
            mut trees => {
                let trees = trees.iter_mut().flat_map(Option::take);
                item_attrs.extend(attr.into_iter().chain(trees).chain(iter));
                return item_attrs.into_iter().collect();
            }
        };
    }
}
